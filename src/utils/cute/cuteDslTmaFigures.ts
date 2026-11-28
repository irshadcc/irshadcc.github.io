// Data for the TMA and tensor-core figures in "The CuTe DSL API, Function by Function"
// (src/content/posts/cute-dsl-api.mdx). Generated from the printed output of TMA kernels run
// with CuTe DSL 4.8.0 on an RTX 5070 Ti (sm_120), and from the wgmma C layout the DSL traces
// for sm_90a. Do not edit by hand. Entries are spread into <Matrix2D> like cuteDslApiFigures.
import { CuteLayout, type IntTuple } from "../../component/Cute/CuteLayout";

interface Figure {
	layout: CuteLayout;
	data?: Record<number, string | number>;
	labels?: Record<number, string>;
}

const fig = (
	shape: IntTuple,
	stride: IntTuple,
	data?: Figure["data"],
	labels?: Figure["labels"],
): Figure => ({
	layout: new CuteLayout(shape, stride),
	data,
	labels,
});

// biome-ignore format: generated data, one figure per line
export const tmaFigures = {
	gmem: fig([6, 12], [12, 1], {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:"pad",11:"pad",12:10,13:11,14:12,15:13,16:14,17:15,18:16,19:17,20:18,21:19,22:"pad",23:"pad",24:20,25:21,26:22,27:23,28:24,29:25,30:26,31:27,32:28,33:29,34:"pad",35:"pad",36:30,37:31,38:32,39:33,40:34,41:35,42:36,43:37,44:38,45:39,46:"pad",47:"pad",48:40,49:41,50:42,51:43,52:44,53:45,54:46,55:47,56:48,57:49,58:"pad",59:"pad",60:50,61:51,62:52,63:53,64:54,65:55,66:56,67:57,68:58,69:59,70:"pad",71:"pad"}, {0:"box (0,0)",1:"box (0,0)",2:"box (0,0)",3:"box (0,0)",4:"box (0,1)",5:"box (0,1)",6:"box (0,1)",7:"box (0,1)",8:"box (0,2)",9:"box (0,2)",10:"padding",11:"padding",12:"box (0,0)",13:"box (0,0)",14:"box (0,0)",15:"box (0,0)",16:"box (0,1)",17:"box (0,1)",18:"box (0,1)",19:"box (0,1)",20:"box (0,2)",21:"box (0,2)",22:"padding",23:"padding",24:"box (0,0)",25:"box (0,0)",26:"box (0,0)",27:"box (0,0)",28:"box (0,1)",29:"box (0,1)",30:"box (0,1)",31:"box (0,1)",32:"box (0,2)",33:"box (0,2)",34:"padding",35:"padding",36:"box (0,0)",37:"box (0,0)",38:"box (0,0)",39:"box (0,0)",40:"box (0,1)",41:"box (0,1)",42:"box (0,1)",43:"box (0,1)",44:"box (0,2)",45:"box (0,2)",46:"padding",47:"padding",48:"box (1,0)",49:"box (1,0)",50:"box (1,0)",51:"box (1,0)",52:"box (1,1)",53:"box (1,1)",54:"box (1,1)",55:"box (1,1)",56:"box (1,2)",57:"box (1,2)",58:"padding",59:"padding",60:"box (1,0)",61:"box (1,0)",62:"box (1,0)",63:"box (1,0)",64:"box (1,1)",65:"box (1,1)",66:"box (1,1)",67:"box (1,1)",68:"box (1,2)",69:"box (1,2)",70:"padding",71:"padding"}),
	box00: fig([4, 4], [4, 1], {0:0,1:1,2:2,3:3,4:10,5:11,6:12,7:13,8:20,9:21,10:22,11:23,12:30,13:31,14:32,15:33}, {0:"(0,0)",1:"(0,1)",2:"(0,2)",3:"(0,3)",4:"(1,0)",5:"(1,1)",6:"(1,2)",7:"(1,3)",8:"(2,0)",9:"(2,1)",10:"(2,2)",11:"(2,3)",12:"(3,0)",13:"(3,1)",14:"(3,2)",15:"(3,3)"}),
	box12: fig([4, 4], [4, 1], {0:48,1:49,2:0,3:0,4:58,5:59,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0}, {0:"(4,8)",1:"(4,9)",2:"past N",3:"past N",4:"(5,8)",5:"(5,9)",6:"past N",7:"past N",8:"past M",9:"past M",10:"past M",11:"past M",12:"past M",13:"past M",14:"past M",15:"past M"}),
	store: fig([4, 12], [12, 1], {0:-7,1:-7,2:-7,3:-7,4:-7,5:-7,6:-7,7:-7,8:100,9:104,10:108,11:112,12:-7,13:-7,14:-7,15:-7,16:-7,17:-7,18:-7,19:-7,20:101,21:105,22:109,23:113,24:-7,25:-7,26:-7,27:-7,28:-7,29:-7,30:-7,31:-7,32:-7,33:-7,34:-7,35:-7,36:-7,37:-7,38:-7,39:-7,40:-7,41:-7,42:-7,43:-7,44:-7,45:-7,46:-7,47:-7}, {0:"kept",1:"kept",2:"kept",3:"kept",4:"kept",5:"kept",6:"kept",7:"kept",8:"in N",9:"in N",10:"past N",11:"past N",12:"kept",13:"kept",14:"kept",15:"kept",16:"kept",17:"kept",18:"kept",19:"kept",20:"in N",21:"in N",22:"past N",23:"past N",24:"kept",25:"kept",26:"kept",27:"kept",28:"kept",29:"kept",30:"kept",31:"kept",32:"kept",33:"kept",34:"kept",35:"kept",36:"kept",37:"kept",38:"kept",39:"kept",40:"kept",41:"kept",42:"kept",43:"kept",44:"kept",45:"kept",46:"kept",47:"kept"}),
	swizzle: fig([8, 8], [8, 1], {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:1,9:0,10:3,11:2,12:5,13:4,14:7,15:6,16:2,17:3,18:0,19:1,20:6,21:7,22:4,23:5,24:3,25:2,26:1,27:0,28:7,29:6,30:5,31:4,32:4,33:5,34:6,35:7,36:0,37:1,38:2,39:3,40:5,41:4,42:7,43:6,44:1,45:0,46:3,47:2,48:6,49:7,50:4,51:5,52:2,53:3,54:0,55:1,56:7,57:6,58:5,59:4,60:3,61:2,62:1,63:0}, {0:"chunk 0",1:"chunk 1",2:"chunk 2",3:"chunk 3",4:"chunk 4",5:"chunk 5",6:"chunk 6",7:"chunk 7",8:"chunk 1",9:"chunk 0",10:"chunk 3",11:"chunk 2",12:"chunk 5",13:"chunk 4",14:"chunk 7",15:"chunk 6",16:"chunk 2",17:"chunk 3",18:"chunk 0",19:"chunk 1",20:"chunk 6",21:"chunk 7",22:"chunk 4",23:"chunk 5",24:"chunk 3",25:"chunk 2",26:"chunk 1",27:"chunk 0",28:"chunk 7",29:"chunk 6",30:"chunk 5",31:"chunk 4",32:"chunk 4",33:"chunk 5",34:"chunk 6",35:"chunk 7",36:"chunk 0",37:"chunk 1",38:"chunk 2",39:"chunk 3",40:"chunk 5",41:"chunk 4",42:"chunk 7",43:"chunk 6",44:"chunk 1",45:"chunk 0",46:"chunk 3",47:"chunk 2",48:"chunk 6",49:"chunk 7",50:"chunk 4",51:"chunk 5",52:"chunk 2",53:"chunk 3",54:"chunk 0",55:"chunk 1",56:"chunk 7",57:"chunk 6",58:"chunk 5",59:"chunk 4",60:"chunk 3",61:"chunk 2",62:"chunk 1",63:"chunk 0"}),
	mcast_both: fig([8, 2], [1, 8], {0:32,1:132,2:232,3:332,4:432,5:532,6:632,7:732,8:32,9:132,10:232,11:332,12:432,13:532,14:632,15:732}, {0:"from CTA 0",1:"from CTA 0",2:"from CTA 0",3:"from CTA 0",4:"from CTA 1",5:"from CTA 1",6:"from CTA 1",7:"from CTA 1",8:"from CTA 0",9:"from CTA 0",10:"from CTA 0",11:"from CTA 0",12:"from CTA 1",13:"from CTA 1",14:"from CTA 1",15:"from CTA 1"}),
	mcast_only0: fig([8, 2], [1, 8], {0:32,1:132,2:232,3:332,4:"\u2013",5:"\u2013",6:"\u2013",7:"\u2013",8:32,9:132,10:232,11:332,12:"\u2013",13:"\u2013",14:"\u2013",15:"\u2013"}, {0:"from CTA 0",1:"from CTA 0",2:"from CTA 0",3:"from CTA 0",4:"never sent",5:"never sent",6:"never sent",7:"never sent",8:"from CTA 0",9:"from CTA 0",10:"from CTA 0",11:"from CTA 0",12:"never sent",13:"never sent",14:"never sent",15:"never sent"}),
	wgmma_c: fig([8, 4], [1, 8], {0:"v0,1",8:"v4,5",16:"v8,9",24:"v12,13",1:"v2,3",9:"v6,7",17:"v10,11",25:"v14,15",2:"v0,1",10:"v4,5",18:"v8,9",26:"v12,13",3:"v2,3",11:"v6,7",19:"v10,11",27:"v14,15",4:"v0,1",12:"v4,5",20:"v8,9",28:"v12,13",5:"v2,3",13:"v6,7",21:"v10,11",29:"v14,15",6:"v0,1",14:"v4,5",22:"v8,9",30:"v12,13",7:"v2,3",15:"v6,7",23:"v10,11",31:"v14,15"}, {0:"warp 0",8:"warp 0",16:"warp 0",24:"warp 0",1:"warp 0",9:"warp 0",17:"warp 0",25:"warp 0",2:"warp 1",10:"warp 1",18:"warp 1",26:"warp 1",3:"warp 1",11:"warp 1",19:"warp 1",27:"warp 1",4:"warp 2",12:"warp 2",20:"warp 2",28:"warp 2",5:"warp 2",13:"warp 2",21:"warp 2",29:"warp 2",6:"warp 3",14:"warp 3",22:"warp 3",30:"warp 3",7:"warp 3",15:"warp 3",23:"warp 3",31:"warp 3"}),
} satisfies Record<string, Figure>;
